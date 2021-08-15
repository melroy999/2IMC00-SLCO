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
import promela.ivar;

/**
 * This is the item provider adapter for a {@link promela.ivar} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class ivarItemProvider
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
	public ivarItemProvider(AdapterFactory adapterFactory) {
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
			addConstantPropertyDescriptor(object);
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
				 getString("_UI_ivar_name_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_ivar_name_feature", "_UI_ivar_type"),
				 PromelaPackage.Literals.IVAR__NAME,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.GENERIC_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This adds a property descriptor for the Constant feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addConstantPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_ivar_constant_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_ivar_constant_feature", "_UI_ivar_type"),
				 PromelaPackage.Literals.IVAR__CONSTANT,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.INTEGRAL_VALUE_IMAGE,
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
			childrenFeatures.add(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT);
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
	 * This returns ivar.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/ivar"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		String label = ((ivar)object).getName();
		return label == null || label.length() == 0 ?
			getString("_UI_ivar_type") :
			getString("_UI_ivar_type") + " " + label;
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

		switch (notification.getFeatureID(ivar.class)) {
			case PromelaPackage.IVAR__NAME:
			case PromelaPackage.IVAR__CONSTANT:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT:
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
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createch_init()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createvarref()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createbin_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createcond_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createlen_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createtimeout_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createnp__expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createenabled_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createpc_value_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createname_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createrun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createnum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createenum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.IVAR__ANY_EXPR_OR_CH_INIT,
				 PromelaFactory.eINSTANCE.createmtype_const()));
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
