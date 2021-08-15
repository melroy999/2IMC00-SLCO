/**
 */
package promela.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.ResourceLocator;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.ItemProviderAdapter;
import org.eclipse.emf.edit.provider.ViewerNotification;

import promela.PromelaFactory;
import promela.PromelaPackage;
import promela.spec;

/**
 * This is the item provider adapter for a {@link promela.spec} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class specItemProvider
	extends ItemProviderAdapter
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
	public specItemProvider(AdapterFactory adapterFactory) {
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

			addNamePropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Name feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addNamePropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_spec_name_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_spec_name_feature", "_UI_spec_type"),
				 PromelaPackage.Literals.SPEC__NAME,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.GENERIC_VALUE_IMAGE,
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
			childrenFeatures.add(PromelaPackage.Literals.SPEC__UTYPES);
			childrenFeatures.add(PromelaPackage.Literals.SPEC__MTYPES);
			childrenFeatures.add(PromelaPackage.Literals.SPEC__ONE_DECLS);
			childrenFeatures.add(PromelaPackage.Literals.SPEC__PROCTYPES);
			childrenFeatures.add(PromelaPackage.Literals.SPEC__INITS);
			childrenFeatures.add(PromelaPackage.Literals.SPEC__NEVERS);
			childrenFeatures.add(PromelaPackage.Literals.SPEC__TRACES);
			childrenFeatures.add(PromelaPackage.Literals.SPEC__PRIMITIVETYPES);
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
	 * This returns spec.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/spec"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		String label = ((spec)object).getName();
		return label == null || label.length() == 0 ?
			getString("_UI_spec_type") :
			getString("_UI_spec_type") + " " + label;
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

		switch (notification.getFeatureID(spec.class)) {
			case PromelaPackage.SPEC__NAME:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case PromelaPackage.SPEC__UTYPES:
			case PromelaPackage.SPEC__MTYPES:
			case PromelaPackage.SPEC__ONE_DECLS:
			case PromelaPackage.SPEC__PROCTYPES:
			case PromelaPackage.SPEC__INITS:
			case PromelaPackage.SPEC__NEVERS:
			case PromelaPackage.SPEC__TRACES:
			case PromelaPackage.SPEC__PRIMITIVETYPES:
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
				(PromelaPackage.Literals.SPEC__UTYPES,
				 PromelaFactory.eINSTANCE.createutype()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.SPEC__MTYPES,
				 PromelaFactory.eINSTANCE.createmtype()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.SPEC__ONE_DECLS,
				 PromelaFactory.eINSTANCE.createone_decl()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.SPEC__PROCTYPES,
				 PromelaFactory.eINSTANCE.createproctype()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.SPEC__INITS,
				 PromelaFactory.eINSTANCE.createinit()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.SPEC__NEVERS,
				 PromelaFactory.eINSTANCE.createnever()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.SPEC__TRACES,
				 PromelaFactory.eINSTANCE.createtrace()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.SPEC__PRIMITIVETYPES,
				 PromelaFactory.eINSTANCE.createprimitivetype()));
	}

	/**
	 * Return the resource locator for this item provider's resources.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public ResourceLocator getResourceLocator() {
		return PromelaEditPlugin.INSTANCE;
	}

}
