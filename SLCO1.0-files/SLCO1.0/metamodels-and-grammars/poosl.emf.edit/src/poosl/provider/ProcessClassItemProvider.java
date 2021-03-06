/**
 */
package poosl.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ViewerNotification;

import poosl.PooslFactory;
import poosl.PooslPackage;
import poosl.ProcessClass;

/**
 * This is the item provider adapter for a {@link poosl.ProcessClass} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class ProcessClassItemProvider
	extends ClassItemProvider
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessClassItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

			addSuperClassPropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Super Class feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addSuperClassPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_ProcessClass_superClass_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_ProcessClass_superClass_feature", "_UI_ProcessClass_type"),
				 PooslPackage.Literals.PROCESS_CLASS__SUPER_CLASS,
				 true,
				 false,
				 true,
				 null,
				 null,
				 null));
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(PooslPackage.Literals.PROCESS_CLASS__VARIABLES);
			childrenFeatures.add(PooslPackage.Literals.PROCESS_CLASS__METHODS);
			childrenFeatures.add(PooslPackage.Literals.PROCESS_CLASS__PARAMETERS);
			childrenFeatures.add(PooslPackage.Literals.PROCESS_CLASS__PORTS);
			childrenFeatures.add(PooslPackage.Literals.PROCESS_CLASS__INITIAL_METHOD_CALL);
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns ProcessClass.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/ProcessClass"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		String label = ((ProcessClass)object).getName();
		return label == null || label.length() == 0 ?
			getString("_UI_ProcessClass_type") :
			getString("_UI_ProcessClass_type") + " " + label;
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(ProcessClass.class)) {
			case PooslPackage.PROCESS_CLASS__VARIABLES:
			case PooslPackage.PROCESS_CLASS__METHODS:
			case PooslPackage.PROCESS_CLASS__PARAMETERS:
			case PooslPackage.PROCESS_CLASS__PORTS:
			case PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.PROCESS_CLASS__VARIABLES,
				 PooslFactory.eINSTANCE.createVariable()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.PROCESS_CLASS__METHODS,
				 PooslFactory.eINSTANCE.createProcessMethod()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.PROCESS_CLASS__PARAMETERS,
				 PooslFactory.eINSTANCE.createParameter()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.PROCESS_CLASS__PORTS,
				 PooslFactory.eINSTANCE.createPort()));

		newChildDescriptors.add
			(createChildParameter
				(PooslPackage.Literals.PROCESS_CLASS__INITIAL_METHOD_CALL,
				 PooslFactory.eINSTANCE.createProcessMethodCall()));
	}

}
